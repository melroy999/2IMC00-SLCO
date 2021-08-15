/**
 */
package slco.impl;

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

import slco.ArgumentType;
import slco.Channel;
import slco.ChannelTypeEnum;
import slco.SlcoPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Channel</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco.impl.ChannelImpl#getArgumentTypes <em>Argument Types</em>}</li>
 *   <li>{@link slco.impl.ChannelImpl#getName <em>Name</em>}</li>
 *   <li>{@link slco.impl.ChannelImpl#getChannelType <em>Channel Type</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public abstract class ChannelImpl extends EObjectImpl implements Channel {
	/**
	 * The cached value of the '{@link #getArgumentTypes() <em>Argument Types</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getArgumentTypes()
	 * @generated
	 * @ordered
	 */
	protected EList<ArgumentType> argumentTypes;

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
	 * The default value of the '{@link #getChannelType() <em>Channel Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getChannelType()
	 * @generated
	 * @ordered
	 */
	protected static final ChannelTypeEnum CHANNEL_TYPE_EDEFAULT = ChannelTypeEnum.ASYNC_LOSSLESS;

	/**
	 * The cached value of the '{@link #getChannelType() <em>Channel Type</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getChannelType()
	 * @generated
	 * @ordered
	 */
	protected ChannelTypeEnum channelType = CHANNEL_TYPE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ChannelImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return SlcoPackage.Literals.CHANNEL;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ArgumentType> getArgumentTypes() {
		if (argumentTypes == null) {
			argumentTypes = new EObjectContainmentEList<ArgumentType>(ArgumentType.class, this, SlcoPackage.CHANNEL__ARGUMENT_TYPES);
		}
		return argumentTypes;
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
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.CHANNEL__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ChannelTypeEnum getChannelType() {
		return channelType;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setChannelType(ChannelTypeEnum newChannelType) {
		ChannelTypeEnum oldChannelType = channelType;
		channelType = newChannelType == null ? CHANNEL_TYPE_EDEFAULT : newChannelType;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.CHANNEL__CHANNEL_TYPE, oldChannelType, channelType));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case SlcoPackage.CHANNEL__ARGUMENT_TYPES:
				return ((InternalEList<?>)getArgumentTypes()).basicRemove(otherEnd, msgs);
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
			case SlcoPackage.CHANNEL__ARGUMENT_TYPES:
				return getArgumentTypes();
			case SlcoPackage.CHANNEL__NAME:
				return getName();
			case SlcoPackage.CHANNEL__CHANNEL_TYPE:
				return getChannelType();
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
			case SlcoPackage.CHANNEL__ARGUMENT_TYPES:
				getArgumentTypes().clear();
				getArgumentTypes().addAll((Collection<? extends ArgumentType>)newValue);
				return;
			case SlcoPackage.CHANNEL__NAME:
				setName((String)newValue);
				return;
			case SlcoPackage.CHANNEL__CHANNEL_TYPE:
				setChannelType((ChannelTypeEnum)newValue);
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
			case SlcoPackage.CHANNEL__ARGUMENT_TYPES:
				getArgumentTypes().clear();
				return;
			case SlcoPackage.CHANNEL__NAME:
				setName(NAME_EDEFAULT);
				return;
			case SlcoPackage.CHANNEL__CHANNEL_TYPE:
				setChannelType(CHANNEL_TYPE_EDEFAULT);
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
			case SlcoPackage.CHANNEL__ARGUMENT_TYPES:
				return argumentTypes != null && !argumentTypes.isEmpty();
			case SlcoPackage.CHANNEL__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case SlcoPackage.CHANNEL__CHANNEL_TYPE:
				return channelType != CHANNEL_TYPE_EDEFAULT;
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
		result.append(", channelType: ");
		result.append(channelType);
		result.append(')');
		return result.toString();
	}

} //ChannelImpl
