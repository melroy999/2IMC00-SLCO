/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import poosl.Parameter;
import poosl.PooslPackage;
import poosl.Port;
import poosl.ProcessClass;
import poosl.ProcessMethod;
import poosl.ProcessMethodCall;
import poosl.Variable;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Process Class</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ProcessClassImpl#getVariables <em>Variables</em>}</li>
 *   <li>{@link poosl.impl.ProcessClassImpl#getMethods <em>Methods</em>}</li>
 *   <li>{@link poosl.impl.ProcessClassImpl#getSuperClass <em>Super Class</em>}</li>
 *   <li>{@link poosl.impl.ProcessClassImpl#getParameters <em>Parameters</em>}</li>
 *   <li>{@link poosl.impl.ProcessClassImpl#getPorts <em>Ports</em>}</li>
 *   <li>{@link poosl.impl.ProcessClassImpl#getInitialMethodCall <em>Initial Method Call</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ProcessClassImpl extends ClassImpl implements ProcessClass {
	/**
	 * The cached value of the '{@link #getVariables() <em>Variables</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVariables()
	 * @generated
	 * @ordered
	 */
	protected EList<Variable> variables;

	/**
	 * The cached value of the '{@link #getMethods() <em>Methods</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMethods()
	 * @generated
	 * @ordered
	 */
	protected EList<ProcessMethod> methods;

	/**
	 * The cached value of the '{@link #getSuperClass() <em>Super Class</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSuperClass()
	 * @generated
	 * @ordered
	 */
	protected ProcessClass superClass;

	/**
	 * The cached value of the '{@link #getParameters() <em>Parameters</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<Parameter> parameters;

	/**
	 * The cached value of the '{@link #getPorts() <em>Ports</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPorts()
	 * @generated
	 * @ordered
	 */
	protected EList<Port> ports;

	/**
	 * The cached value of the '{@link #getInitialMethodCall() <em>Initial Method Call</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInitialMethodCall()
	 * @generated
	 * @ordered
	 */
	protected ProcessMethodCall initialMethodCall;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ProcessClassImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.PROCESS_CLASS;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Variable> getVariables() {
		if (variables == null) {
			variables = new EObjectContainmentEList<Variable>(Variable.class, this, PooslPackage.PROCESS_CLASS__VARIABLES);
		}
		return variables;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ProcessMethod> getMethods() {
		if (methods == null) {
			methods = new EObjectContainmentEList<ProcessMethod>(ProcessMethod.class, this, PooslPackage.PROCESS_CLASS__METHODS);
		}
		return methods;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessClass getSuperClass() {
		if (superClass != null && superClass.eIsProxy()) {
			InternalEObject oldSuperClass = (InternalEObject)superClass;
			superClass = (ProcessClass)eResolveProxy(oldSuperClass);
			if (superClass != oldSuperClass) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.PROCESS_CLASS__SUPER_CLASS, oldSuperClass, superClass));
			}
		}
		return superClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessClass basicGetSuperClass() {
		return superClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSuperClass(ProcessClass newSuperClass) {
		ProcessClass oldSuperClass = superClass;
		superClass = newSuperClass;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.PROCESS_CLASS__SUPER_CLASS, oldSuperClass, superClass));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Parameter> getParameters() {
		if (parameters == null) {
			parameters = new EObjectContainmentEList<Parameter>(Parameter.class, this, PooslPackage.PROCESS_CLASS__PARAMETERS);
		}
		return parameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Port> getPorts() {
		if (ports == null) {
			ports = new EObjectContainmentEList<Port>(Port.class, this, PooslPackage.PROCESS_CLASS__PORTS);
		}
		return ports;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessMethodCall getInitialMethodCall() {
		return initialMethodCall;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetInitialMethodCall(ProcessMethodCall newInitialMethodCall, NotificationChain msgs) {
		ProcessMethodCall oldInitialMethodCall = initialMethodCall;
		initialMethodCall = newInitialMethodCall;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL, oldInitialMethodCall, newInitialMethodCall);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setInitialMethodCall(ProcessMethodCall newInitialMethodCall) {
		if (newInitialMethodCall != initialMethodCall) {
			NotificationChain msgs = null;
			if (initialMethodCall != null)
				msgs = ((InternalEObject)initialMethodCall).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL, null, msgs);
			if (newInitialMethodCall != null)
				msgs = ((InternalEObject)newInitialMethodCall).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL, null, msgs);
			msgs = basicSetInitialMethodCall(newInitialMethodCall, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL, newInitialMethodCall, newInitialMethodCall));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.PROCESS_CLASS__VARIABLES:
				return ((InternalEList<?>)getVariables()).basicRemove(otherEnd, msgs);
			case PooslPackage.PROCESS_CLASS__METHODS:
				return ((InternalEList<?>)getMethods()).basicRemove(otherEnd, msgs);
			case PooslPackage.PROCESS_CLASS__PARAMETERS:
				return ((InternalEList<?>)getParameters()).basicRemove(otherEnd, msgs);
			case PooslPackage.PROCESS_CLASS__PORTS:
				return ((InternalEList<?>)getPorts()).basicRemove(otherEnd, msgs);
			case PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL:
				return basicSetInitialMethodCall(null, msgs);
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
			case PooslPackage.PROCESS_CLASS__VARIABLES:
				return getVariables();
			case PooslPackage.PROCESS_CLASS__METHODS:
				return getMethods();
			case PooslPackage.PROCESS_CLASS__SUPER_CLASS:
				if (resolve) return getSuperClass();
				return basicGetSuperClass();
			case PooslPackage.PROCESS_CLASS__PARAMETERS:
				return getParameters();
			case PooslPackage.PROCESS_CLASS__PORTS:
				return getPorts();
			case PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL:
				return getInitialMethodCall();
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
			case PooslPackage.PROCESS_CLASS__VARIABLES:
				getVariables().clear();
				getVariables().addAll((Collection<? extends Variable>)newValue);
				return;
			case PooslPackage.PROCESS_CLASS__METHODS:
				getMethods().clear();
				getMethods().addAll((Collection<? extends ProcessMethod>)newValue);
				return;
			case PooslPackage.PROCESS_CLASS__SUPER_CLASS:
				setSuperClass((ProcessClass)newValue);
				return;
			case PooslPackage.PROCESS_CLASS__PARAMETERS:
				getParameters().clear();
				getParameters().addAll((Collection<? extends Parameter>)newValue);
				return;
			case PooslPackage.PROCESS_CLASS__PORTS:
				getPorts().clear();
				getPorts().addAll((Collection<? extends Port>)newValue);
				return;
			case PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL:
				setInitialMethodCall((ProcessMethodCall)newValue);
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
			case PooslPackage.PROCESS_CLASS__VARIABLES:
				getVariables().clear();
				return;
			case PooslPackage.PROCESS_CLASS__METHODS:
				getMethods().clear();
				return;
			case PooslPackage.PROCESS_CLASS__SUPER_CLASS:
				setSuperClass((ProcessClass)null);
				return;
			case PooslPackage.PROCESS_CLASS__PARAMETERS:
				getParameters().clear();
				return;
			case PooslPackage.PROCESS_CLASS__PORTS:
				getPorts().clear();
				return;
			case PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL:
				setInitialMethodCall((ProcessMethodCall)null);
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
			case PooslPackage.PROCESS_CLASS__VARIABLES:
				return variables != null && !variables.isEmpty();
			case PooslPackage.PROCESS_CLASS__METHODS:
				return methods != null && !methods.isEmpty();
			case PooslPackage.PROCESS_CLASS__SUPER_CLASS:
				return superClass != null;
			case PooslPackage.PROCESS_CLASS__PARAMETERS:
				return parameters != null && !parameters.isEmpty();
			case PooslPackage.PROCESS_CLASS__PORTS:
				return ports != null && !ports.isEmpty();
			case PooslPackage.PROCESS_CLASS__INITIAL_METHOD_CALL:
				return initialMethodCall != null;
		}
		return super.eIsSet(featureID);
	}

} //ProcessClassImpl
