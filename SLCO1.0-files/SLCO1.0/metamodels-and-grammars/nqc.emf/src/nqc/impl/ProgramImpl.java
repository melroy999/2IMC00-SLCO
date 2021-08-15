/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import java.util.Collection;

import nqc.Function;
import nqc.NqcPackage;
import nqc.Program;
import nqc.Subroutine;
import nqc.Task;
import nqc.Variable;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Program</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.ProgramImpl#getName <em>Name</em>}</li>
 *   <li>{@link nqc.impl.ProgramImpl#getTasks <em>Tasks</em>}</li>
 *   <li>{@link nqc.impl.ProgramImpl#getFunctions <em>Functions</em>}</li>
 *   <li>{@link nqc.impl.ProgramImpl#getSubroutines <em>Subroutines</em>}</li>
 *   <li>{@link nqc.impl.ProgramImpl#getGlobalVariables <em>Global Variables</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ProgramImpl extends EObjectImpl implements Program {
	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;

	/**
	 * The cached value of the '{@link #getTasks() <em>Tasks</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTasks()
	 * @generated
	 * @ordered
	 */
	protected EList<Task> tasks;

	/**
	 * The cached value of the '{@link #getFunctions() <em>Functions</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getFunctions()
	 * @generated
	 * @ordered
	 */
	protected EList<Function> functions;

	/**
	 * The cached value of the '{@link #getSubroutines() <em>Subroutines</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSubroutines()
	 * @generated
	 * @ordered
	 */
	protected EList<Subroutine> subroutines;

	/**
	 * The cached value of the '{@link #getGlobalVariables() <em>Global Variables</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getGlobalVariables()
	 * @generated
	 * @ordered
	 */
	protected EList<Variable> globalVariables;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ProgramImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getProgram();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.PROGRAM__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Task> getTasks() {
		if (tasks == null) {
			tasks = new EObjectContainmentEList<Task>(Task.class, this, NqcPackage.PROGRAM__TASKS);
		}
		return tasks;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Function> getFunctions() {
		if (functions == null) {
			functions = new EObjectContainmentEList<Function>(Function.class, this, NqcPackage.PROGRAM__FUNCTIONS);
		}
		return functions;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Subroutine> getSubroutines() {
		if (subroutines == null) {
			subroutines = new EObjectContainmentEList<Subroutine>(Subroutine.class, this, NqcPackage.PROGRAM__SUBROUTINES);
		}
		return subroutines;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Variable> getGlobalVariables() {
		if (globalVariables == null) {
			globalVariables = new EObjectContainmentEList<Variable>(Variable.class, this, NqcPackage.PROGRAM__GLOBAL_VARIABLES);
		}
		return globalVariables;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case NqcPackage.PROGRAM__TASKS:
				return ((InternalEList<?>)getTasks()).basicRemove(otherEnd, msgs);
			case NqcPackage.PROGRAM__FUNCTIONS:
				return ((InternalEList<?>)getFunctions()).basicRemove(otherEnd, msgs);
			case NqcPackage.PROGRAM__SUBROUTINES:
				return ((InternalEList<?>)getSubroutines()).basicRemove(otherEnd, msgs);
			case NqcPackage.PROGRAM__GLOBAL_VARIABLES:
				return ((InternalEList<?>)getGlobalVariables()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.PROGRAM__NAME:
				return getName();
			case NqcPackage.PROGRAM__TASKS:
				return getTasks();
			case NqcPackage.PROGRAM__FUNCTIONS:
				return getFunctions();
			case NqcPackage.PROGRAM__SUBROUTINES:
				return getSubroutines();
			case NqcPackage.PROGRAM__GLOBAL_VARIABLES:
				return getGlobalVariables();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case NqcPackage.PROGRAM__NAME:
				setName((String)newValue);
				return;
			case NqcPackage.PROGRAM__TASKS:
				getTasks().clear();
				getTasks().addAll((Collection<? extends Task>)newValue);
				return;
			case NqcPackage.PROGRAM__FUNCTIONS:
				getFunctions().clear();
				getFunctions().addAll((Collection<? extends Function>)newValue);
				return;
			case NqcPackage.PROGRAM__SUBROUTINES:
				getSubroutines().clear();
				getSubroutines().addAll((Collection<? extends Subroutine>)newValue);
				return;
			case NqcPackage.PROGRAM__GLOBAL_VARIABLES:
				getGlobalVariables().clear();
				getGlobalVariables().addAll((Collection<? extends Variable>)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case NqcPackage.PROGRAM__NAME:
				setName(NAME_EDEFAULT);
				return;
			case NqcPackage.PROGRAM__TASKS:
				getTasks().clear();
				return;
			case NqcPackage.PROGRAM__FUNCTIONS:
				getFunctions().clear();
				return;
			case NqcPackage.PROGRAM__SUBROUTINES:
				getSubroutines().clear();
				return;
			case NqcPackage.PROGRAM__GLOBAL_VARIABLES:
				getGlobalVariables().clear();
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case NqcPackage.PROGRAM__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case NqcPackage.PROGRAM__TASKS:
				return tasks != null && !tasks.isEmpty();
			case NqcPackage.PROGRAM__FUNCTIONS:
				return functions != null && !functions.isEmpty();
			case NqcPackage.PROGRAM__SUBROUTINES:
				return subroutines != null && !subroutines.isEmpty();
			case NqcPackage.PROGRAM__GLOBAL_VARIABLES:
				return globalVariables != null && !globalVariables.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuffer result = new StringBuffer(super.toString());
		result.append(" (Name: ");
		result.append(name);
		result.append(')');
		return result.toString();
	}

} //ProgramImpl
