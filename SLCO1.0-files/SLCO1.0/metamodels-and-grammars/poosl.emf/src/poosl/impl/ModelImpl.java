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
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import poosl.Channel;
import poosl.Model;
import poosl.PooslPackage;
import poosl.PrimitiveType;
import poosl.ProcessInstance;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Model</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ModelImpl#getName <em>Name</em>}</li>
 *   <li>{@link poosl.impl.ModelImpl#getClasses <em>Classes</em>}</li>
 *   <li>{@link poosl.impl.ModelImpl#getProcessInstances <em>Process Instances</em>}</li>
 *   <li>{@link poosl.impl.ModelImpl#getChannels <em>Channels</em>}</li>
 *   <li>{@link poosl.impl.ModelImpl#getPrimitiveTypes <em>Primitive Types</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ModelImpl extends EObjectImpl implements Model {
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
	 * The cached value of the '{@link #getClasses() <em>Classes</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getClasses()
	 * @generated
	 * @ordered
	 */
	protected EList<poosl.Class> classes;

	/**
	 * The cached value of the '{@link #getProcessInstances() <em>Process Instances</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getProcessInstances()
	 * @generated
	 * @ordered
	 */
	protected EList<ProcessInstance> processInstances;

	/**
	 * The cached value of the '{@link #getChannels() <em>Channels</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getChannels()
	 * @generated
	 * @ordered
	 */
	protected EList<Channel> channels;

	/**
	 * The cached value of the '{@link #getPrimitiveTypes() <em>Primitive Types</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrimitiveTypes()
	 * @generated
	 * @ordered
	 */
	protected EList<PrimitiveType> primitiveTypes;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ModelImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.MODEL;
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
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.MODEL__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<poosl.Class> getClasses() {
		if (classes == null) {
			classes = new EObjectContainmentEList<poosl.Class>(poosl.Class.class, this, PooslPackage.MODEL__CLASSES);
		}
		return classes;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ProcessInstance> getProcessInstances() {
		if (processInstances == null) {
			processInstances = new EObjectContainmentEList<ProcessInstance>(ProcessInstance.class, this, PooslPackage.MODEL__PROCESS_INSTANCES);
		}
		return processInstances;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Channel> getChannels() {
		if (channels == null) {
			channels = new EObjectContainmentEList<Channel>(Channel.class, this, PooslPackage.MODEL__CHANNELS);
		}
		return channels;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<PrimitiveType> getPrimitiveTypes() {
		if (primitiveTypes == null) {
			primitiveTypes = new EObjectContainmentEList<PrimitiveType>(PrimitiveType.class, this, PooslPackage.MODEL__PRIMITIVE_TYPES);
		}
		return primitiveTypes;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.MODEL__CLASSES:
				return ((InternalEList<?>)getClasses()).basicRemove(otherEnd, msgs);
			case PooslPackage.MODEL__PROCESS_INSTANCES:
				return ((InternalEList<?>)getProcessInstances()).basicRemove(otherEnd, msgs);
			case PooslPackage.MODEL__CHANNELS:
				return ((InternalEList<?>)getChannels()).basicRemove(otherEnd, msgs);
			case PooslPackage.MODEL__PRIMITIVE_TYPES:
				return ((InternalEList<?>)getPrimitiveTypes()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.MODEL__NAME:
				return getName();
			case PooslPackage.MODEL__CLASSES:
				return getClasses();
			case PooslPackage.MODEL__PROCESS_INSTANCES:
				return getProcessInstances();
			case PooslPackage.MODEL__CHANNELS:
				return getChannels();
			case PooslPackage.MODEL__PRIMITIVE_TYPES:
				return getPrimitiveTypes();
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
			case PooslPackage.MODEL__NAME:
				setName((String)newValue);
				return;
			case PooslPackage.MODEL__CLASSES:
				getClasses().clear();
				getClasses().addAll((Collection<? extends poosl.Class>)newValue);
				return;
			case PooslPackage.MODEL__PROCESS_INSTANCES:
				getProcessInstances().clear();
				getProcessInstances().addAll((Collection<? extends ProcessInstance>)newValue);
				return;
			case PooslPackage.MODEL__CHANNELS:
				getChannels().clear();
				getChannels().addAll((Collection<? extends Channel>)newValue);
				return;
			case PooslPackage.MODEL__PRIMITIVE_TYPES:
				getPrimitiveTypes().clear();
				getPrimitiveTypes().addAll((Collection<? extends PrimitiveType>)newValue);
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
			case PooslPackage.MODEL__NAME:
				setName(NAME_EDEFAULT);
				return;
			case PooslPackage.MODEL__CLASSES:
				getClasses().clear();
				return;
			case PooslPackage.MODEL__PROCESS_INSTANCES:
				getProcessInstances().clear();
				return;
			case PooslPackage.MODEL__CHANNELS:
				getChannels().clear();
				return;
			case PooslPackage.MODEL__PRIMITIVE_TYPES:
				getPrimitiveTypes().clear();
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
			case PooslPackage.MODEL__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case PooslPackage.MODEL__CLASSES:
				return classes != null && !classes.isEmpty();
			case PooslPackage.MODEL__PROCESS_INSTANCES:
				return processInstances != null && !processInstances.isEmpty();
			case PooslPackage.MODEL__CHANNELS:
				return channels != null && !channels.isEmpty();
			case PooslPackage.MODEL__PRIMITIVE_TYPES:
				return primitiveTypes != null && !primitiveTypes.isEmpty();
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
		result.append(" (name: ");
		result.append(name);
		result.append(')');
		return result.toString();
	}

} //ModelImpl
